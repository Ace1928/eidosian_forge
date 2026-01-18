from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def softblock_workers(self):
    """
        Softblock workers if necessary.
        """
    if not self.opt['is_sandbox'] and self.opt['softblock_list_path'] is not None:
        softblock_list = set()
        with open(self.opt['softblock_list_path']) as f:
            for line in f:
                softblock_list.add(line.strip())
        print(f'Will softblock {len(softblock_list):d} workers.')
        for w in softblock_list:
            try:
                print('Soft Blocking {}\n'.format(w))
                self.manager.soft_block_worker(w)
            except Exception as e:
                print(f'Did not soft block worker {w}: {e}')
            time.sleep(0.1)