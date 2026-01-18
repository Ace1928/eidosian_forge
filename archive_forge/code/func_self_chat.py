from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import math
import json
import random
def self_chat(opt):
    random.seed(opt['seed'])
    partner = opt['partner_model_file']
    partner_opt_file = opt.get('partner_opt_file')
    agent1 = create_agent(opt, requireModelExists=True)
    agent1.opt.log('Agent 1 Opt')
    if partner is None:
        agent2 = agent1.clone()
    else:
        if partner_opt_file:
            print(f'WARNING: Loading override opts from: {partner_opt_file}')
            with open(partner_opt_file) as f:
                partner_opt = json.load(f)
        else:
            partner_opt = {}
        partner_opt['interactive_mode'] = opt.get('interactive_mode', True)
        print(f'WARNING: Setting partner interactive mode to: {partner_opt['interactive_mode']}')
        agent2 = create_agent_from_model_file(partner, partner_opt)
        agent2.opt.log('Agent 2 Opt')
    agent1.id = agent1.id + '_1'
    agent2.id = agent2.id + '_2'
    model_id = agent1.id + '_' + agent2.id
    world = create_task(opt, user_agents=[agent1, agent2])
    logger = WorldLogger(opt)
    log_time = TimeLogger()
    for i in range(opt['num_self_chats']):
        _run_self_chat_episode(opt, world, logger)
        report = world.report()
        text, report = log_time.log(i + 1, opt['num_self_chats'], report)
        logging.info(text)
    if opt['outfile'] is None:
        outfile = '/tmp/{}_selfchat'.format(model_id)
    else:
        outfile = opt['outfile']
    if opt['save_format'] == 'conversations' and hasattr(world, 'write'):
        world.write(logger, outfile)
    else:
        logger.write(outfile, world, opt['save_format'])
    return logger.get_logs()