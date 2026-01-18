from parlai.core.agents import create_agent
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.core.params import ParlaiParser
from parlai.utils.misc import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import (
from task_config import task_config
from parlai.core.dict import DictionaryAgent
import os
import copy
import tqdm
import pickle
import parlai.core.build_data as build_data
from urllib.parse import unquote
def setup_personas_with_wiki_links(opt):
    fname = 'personas_with_wiki_links.txt'
    file_path = '{}/{}'.format(os.getcwd(), fname)
    if not os.path.exists(file_path):
        url = 'http://parl.ai/downloads/wizard_of_wikipedia/' + fname
        build_data.download(url, os.getcwd(), fname)