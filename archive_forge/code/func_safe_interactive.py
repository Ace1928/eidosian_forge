from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import parlai.utils.logging as logging
import random
def safe_interactive(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    human_agent = SafeLocalHumanAgent(opt)
    world = create_task(opt, [human_agent, agent])
    while True:
        world.parley()
        bot_act = world.get_acts()[-1]
        if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
            agent.reset()
        if opt.get('display_examples'):
            print('---')
            print(world.display())
        if world.epoch_done():
            logging.info('epoch done')
            break