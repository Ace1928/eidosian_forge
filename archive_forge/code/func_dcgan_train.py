import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import argparse
import os
from filelock import FileLock
import tempfile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from ray.tune.examples.pbt_dcgan_mnist.common import (
def dcgan_train(config):
    use_cuda = config.get('use_gpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    netG = Generator().to(device)
    netG.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.get('lr', 0.01), betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.get('lr', 0.01), betas=(beta1, 0.999))
    with FileLock(os.path.expanduser('~/ray_results/.data.lock')):
        dataloader = get_data_loader()
    step = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        netD.load_state_dict(checkpoint_dict['netDmodel'])
        netG.load_state_dict(checkpoint_dict['netGmodel'])
        optimizerD.load_state_dict(checkpoint_dict['optimD'])
        optimizerG.load_state_dict(checkpoint_dict['optimG'])
        last_step = checkpoint_dict['step']
        step = last_step + 1
        if 'netD_lr' in config:
            for param_group in optimizerD.param_groups:
                param_group['lr'] = config['netD_lr']
        if 'netG_lr' in config:
            for param_group in optimizerG.param_groups:
                param_group['lr'] = config['netG_lr']
    while True:
        lossG, lossD, is_score = train_func(netD, netG, optimizerG, optimizerD, criterion, dataloader, step, device, config['mnist_model_ref'])
        metrics = {'lossg': lossG, 'lossd': lossD, 'is_score': is_score}
        if step % config['checkpoint_interval'] == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save({'netDmodel': netD.state_dict(), 'netGmodel': netG.state_dict(), 'optimD': optimizerD.state_dict(), 'optimG': optimizerG.state_dict(), 'step': step}, os.path.join(tmpdir, 'checkpoint.pt'))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            train.report(metrics)
        step += 1