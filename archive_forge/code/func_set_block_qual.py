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
def set_block_qual(self, task_group_id: str):
    """
        Set block qualification if necessary.

        :param task_group_id:
            task id used to set block qualification, if necessary.
        """
    if self.opt['block_on_onboarding_fail'] and self.opt['block_qualification'] is None:
        self.opt['block_qualification'] = task_group_id
        warn_once('No block_qualification set in opt, automatically creating new qualification {}'.format(task_group_id))