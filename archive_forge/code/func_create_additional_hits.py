import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def create_additional_hits(self, num_hits, qualifications=None):
    """
        Handle creation for a specific number of hits/assignments Put created HIT ids
        into the hit_id_list.
        """
    shared_utils.print_and_log(logging.INFO, 'Creating {} hits...'.format(num_hits))
    qualifications = self.get_qualification_list(qualifications)
    self.opt['assignment_duration_in_seconds'] = self.opt.get('assignment_duration_in_seconds', 30 * 60)
    hit_type_id = mturk_utils.create_hit_type(hit_title=self.opt['hit_title'], hit_description='{} (ID: {})'.format(self.opt['hit_description'], self.task_group_id), hit_keywords=self.opt['hit_keywords'], hit_reward=self.opt['reward'], assignment_duration_in_seconds=self.opt.get('assignment_duration_in_seconds', 30 * 60), is_sandbox=self.opt['is_sandbox'], qualifications=qualifications, auto_approve_delay=self.auto_approve_delay)
    mturk_chat_url = '{}/chat_index?task_group_id={}'.format(self.server_url, self.task_group_id)
    shared_utils.print_and_log(logging.INFO, mturk_chat_url)
    mturk_page_url = None
    if self.topic_arn is not None:
        mturk_utils.subscribe_to_hits(hit_type_id, self.is_sandbox, self.topic_arn)
    for _i in range(num_hits):
        mturk_page_url, hit_id, mturk_response = mturk_utils.create_hit_with_hit_type(opt=self.opt, page_url=mturk_chat_url, hit_type_id=hit_type_id, num_assignments=1, is_sandbox=self.is_sandbox)
        if self.db_logger is not None:
            self.db_logger.log_hit_status(mturk_response)
        self.hit_id_list.append(hit_id)
        self.all_hit_ids.append(hit_id)
    return mturk_page_url