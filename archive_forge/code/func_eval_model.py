from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import (
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import json
import random
from parlai.utils.distributed import (
def eval_model(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :return: the final result of calling report()
    """
    random.seed(42)
    if 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']:
        raise ValueError('You should use --datatype train:evalmode if you want to evaluate on the training set.')
    if opt['save_world_logs'] and (not opt['report_filename']):
        raise RuntimeError('In order to save model replies, please specify the save path with --report-filename')
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    tasks = opt['task'].split(',')
    reports = []
    for task in tasks:
        task_report = _eval_single_world(opt, agent, task)
        reports.append(task_report)
    report = aggregate_named_reports(dict(zip(tasks, reports)), micro_average=opt.get('aggregate_micro', False))
    print_announcements(opt)
    logging.info(f'Finished evaluating tasks {tasks} using datatype {opt.get('datatype')}')
    print(nice_report(report))
    _save_eval_stats(opt, report)
    return report