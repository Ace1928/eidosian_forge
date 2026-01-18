import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from .model import LORA_CONFIG
def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f' aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet', shell=True)
    elif 'world/14b' in ff or 'world/7b' in ff:
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f' aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet', shell=True)
    elif 'deepspeed_stage_3' in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)