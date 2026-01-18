from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.repeat_query.repeat_query import RepeatQueryAgent
import parlai.utils.logging as logging
import random
import cProfile
import io
import pstats
def profile_interactive(opt):
    agent = create_agent(opt, requireModelExists=True)
    human_agent = RepeatQueryAgent(opt)
    world = create_task(opt, [human_agent, agent])
    agent.opt.log()
    pr = cProfile.Profile()
    pr.enable()
    cnt = 0
    while True:
        world.parley()
        if opt.get('display_examples'):
            print('---')
            print(world.display())
        cnt += 1
        if cnt >= opt.get('num_examples', 100):
            break
        if world.epoch_done():
            logging.info('epoch done')
            break
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())