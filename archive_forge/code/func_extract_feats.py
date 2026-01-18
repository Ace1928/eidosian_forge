import copy
import tqdm
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
def extract_feats(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('extract_feats should be passed opt not parser')
        opt = opt.parse_args()
    opt = copy.deepcopy(opt)
    dt = opt['datatype'].split(':')[0] + ':ordered'
    opt['datatype'] = dt
    opt['no_cuda'] = False
    opt['gpu'] = 0
    opt['num_epochs'] = 1
    opt['num_load_threads'] = 20
    opt.log()
    logging.info('Loading Images')
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    total_exs = world.num_examples()
    pbar = tqdm.tqdm(unit='ex', total=total_exs)
    while not world.epoch_done():
        world.parley()
        pbar.update()
    pbar.close()
    logging.info('Finished extracting images')