from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
import random
def display_model(opt):
    random.seed(42)
    agent = create_agent(opt)
    world = create_task(opt, agent)
    agent.opt.log()
    turn = 0
    with world:
        for _k in range(int(opt['num_examples'])):
            world.parley()
            if opt['verbose']:
                print(world.display() + '\n~~')
            else:
                simple_display(opt, world, turn)
            turn += 1
            if world.get_acts()[0]['episode_done']:
                turn = 0
            if world.epoch_done():
                logging.info('epoch done')
                turn = 0
                break