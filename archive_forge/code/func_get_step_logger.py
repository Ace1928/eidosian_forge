import itertools
import logging
from torch.hub import _Faketqdm, tqdm
def get_step_logger(logger):
    if not disable_progress:
        pbar.update(1)
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f'{logger.name}')
    step = next(_step_counter)

    def log(level, msg):
        logger.log(level, 'Step %s: %s', step, msg)
    return log