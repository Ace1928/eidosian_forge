import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger
    model = trainer.model
    args = trainer.args
    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)
    ds_logger.setLevel(args.get_process_log_level())
    if inference:
        if not hf_deepspeed_config.is_zero3():
            raise ValueError('ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config')
        hf_deepspeed_config.del_config_sub_tree('optimizer')
        hf_deepspeed_config.del_config_sub_tree('lr_scheduler')
        optimizer, lr_scheduler = (None, None)
        model_parameters = None
    else:
        trainer.optimizer = None
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters)
    return (optimizer, lr_scheduler)