from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.strings import colorize
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import random
def display_data(opt):
    if 'ordered' not in opt['datatype'] and 'train' in opt['datatype']:
        opt['datatype'] = f'{opt['datatype']}:ordered'
    opt.log()
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    turn = 0
    for _ in range(opt['num_examples']):
        world.parley()
        if opt['display_verbose']:
            print(world.display() + '\n~~')
        else:
            simple_display(opt, world, turn)
            turn += 1
            if world.get_acts()[0]['episode_done']:
                turn = 0
        if world.epoch_done():
            logging.info('epoch done')
            break
    try:
        logging.info(f'loaded {world.num_episodes()} episodes with a total of {world.num_examples()} examples')
    except Exception:
        pass