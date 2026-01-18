from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.legacy_2018.mturk_manager import MTurkManager
import parlai.mturk.core.legacy_2018.mturk_utils as mturk_utils
from worlds import ControllableDialogEval, PersonasGenerator, PersonaAssignWorld
from task_config import task_config
import model_configs as mcf
from threading import Lock
import gc
import datetime
import json
import os
import sys
import copy
import random
import pprint
from parlai.utils.logging import ParlaiLogger, INFO
def run_conversation(mturk_manager, opt, workers):
    conv_idx = mturk_manager.conversation_index
    assert len(workers) == 1
    worker_id = workers[0].worker_id
    lock.acquire()
    if worker_id not in worker_models_seen:
        worker_models_seen[worker_id] = set()
    print('MODELCOUNTS:')
    print(pprint.pformat(model_counts))
    logger.info('MODELCOUNTS\n' + pprint.pformat(model_counts))
    model_options = [(model_counts[setup_name] + 10 * random.random(), setup_name) for setup_name in SETTINGS_TO_RUN if setup_name not in worker_models_seen[worker_id]]
    if not model_options:
        lock.release()
        logger.error('Worker {} already finished all settings! Returning none'.format(worker_id))
        return None
    _, model_choice = min(model_options)
    worker_models_seen[worker_id].add(model_choice)
    model_counts[model_choice] += 1
    lock.release()
    world = ControllableDialogEval(opt=model_opts[model_choice], agents=workers, num_turns=start_opt['num_turns'], max_resp_time=start_opt['max_resp_time'], model_agent_opt=model_share_params[model_choice], world_tag='conversation t_{}'.format(conv_idx), agent_timeout_shutdown=opt['ag_shutdown_time'], model_config=model_choice)
    world.reset_random()
    while not world.episode_done():
        world.parley()
    world.save_data()
    lock.acquire()
    if not world.convo_finished:
        model_counts[model_choice] -= 1
        worker_models_seen[worker_id].remove(model_choice)
    lock.release()
    world.shutdown()
    gc.collect()