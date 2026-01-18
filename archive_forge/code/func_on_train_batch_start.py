import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from .model import LORA_CONFIG
def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    args = self.args
    real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
    w_step = args.warmup_steps
    if args.lr_final == args.lr_init or args.epoch_count == 0:
        lr = args.lr_init
    else:
        decay_step = real_step - args.my_pile_edecay * args.epoch_steps
        decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
        progress = (decay_step - w_step + 1) / (decay_total - w_step)
        progress = min(1, max(0, progress))
        if args.lr_final == 0 or args.lr_init == 0:
            lr = args.lr_init + (args.lr_final - args.lr_init) * progress
        else:
            lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
    if args.my_exit_tokens != 0:
        real_tokens = real_step * args.ctx_len * args.real_bsz
        warmup_tokens = w_step * args.ctx_len * args.real_bsz
        progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
        progress = max(0, min(1, progress))
        lr_final_factor = args.lr_final / args.lr_init
        lr_mult = 0.5 + lr_final_factor / 2 + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
        if args.my_exit_tokens > 0:
            lr = args.lr_init * lr_mult
        else:
            lr = (lr + args.lr_init * lr_mult) / 2
        if progress >= 1:
            if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
                my_save(args, trainer, pl_module.state_dict(), f'{args.proj_dir}/rwkv-final.pth')
                exit(0)
    if trainer.global_step < w_step:
        lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
    if args.weight_decay_final > 0:
        wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
    else:
        wd_now = args.weight_decay
    for param_group in trainer.optimizers[0].param_groups:
        if param_group['weight_decay'] > 0:
            param_group['weight_decay'] = wd_now
        if args.layerwise_lr > 0:
            param_group['lr'] = lr * param_group['my_lr_scale']
        else:
            param_group['lr'] = lr
    trainer.my_lr = lr
    trainer.my_wd = wd_now
    if trainer.global_step == 0:
        if trainer.is_global_zero:
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            trainer.my_log = open(args.proj_dir + '/train_log.txt', 'a')
            trainer.my_log.write(f'NEW RUN {args.my_timestamp}\n{vars(self.args)}\n')
            try:
                print(f'\n{trainer.strategy.config}\n')
                trainer.my_log.write(f'{trainer.strategy.config}\n')
            except:
                pass
            trainer.my_log.flush()
            if len(args.wandb) > 0:
                print('Login to wandb...')
                import wandb
                wandb.init(project=args.wandb, name=args.run_name + ' ' + args.my_timestamp, config=args, save_code=False)
                trainer.my_wandb = wandb