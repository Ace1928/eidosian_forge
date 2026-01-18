from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
import random
import tempfile
def build_cands(opt):
    opt.log()
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    if opt['outfile'] is None:
        outfile = tempfile.mkstemp(prefix='{}_{}_'.format(opt['task'], opt['datatype']), suffix='.txt')[1]
    else:
        outfile = opt['outfile']
    if opt.get('num_examples', -1) == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']
    log_timer = TimeLogger()
    logging.info(f'Starting to build candidates from task.. (ex: {num_examples})')
    logging.info(f'Saving output to {outfile}')
    cands = set()
    for _ in range(num_examples):
        world.parley()
        acts = world.get_acts()[0]
        if isinstance(acts, dict):
            acts = [acts]
        for a in acts:
            candidate = a.get('labels', a.get('eval_labels', None))
            if candidate is not None:
                candidate = candidate[0]
                cands.add(candidate)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            logging.info(text)
        if world.epoch_done():
            logging.info('epoch done')
            break
    fw = open(outfile, 'w')
    fw.write('\n'.join(cands))
    fw.close()