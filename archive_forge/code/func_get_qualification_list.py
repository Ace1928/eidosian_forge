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
def get_qualification_list(self, qualifications=None):
    if self.qualifications is not None:
        return self.qualifications.copy()
    if qualifications is None:
        qualifications = []
    if not self.is_sandbox and (not self.is_test):
        try:
            import parlai_internal.mturk.configs as local_configs
            qualifications = local_configs.set_default_qualifications(qualifications)
        except Exception:
            pass
    if self.opt['disconnect_qualification'] is not None:
        block_qual_id = mturk_utils.find_or_create_qualification(self.opt['disconnect_qualification'], 'A soft ban from using a ParlAI-created HIT due to frequent disconnects from conversations, leading to negative experiences for other Turkers and for the requester.', self.is_sandbox)
        assert block_qual_id is not None, 'Hits could not be created as disconnect qualification could not be acquired. Shutting down server.'
        qualifications.append({'QualificationTypeId': block_qual_id, 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'})
    if self.opt['block_qualification'] is not None:
        block_qual_id = mturk_utils.find_or_create_qualification(self.opt['block_qualification'], 'A soft ban from this ParlAI-created HIT at the requesters discretion. Generally used to restrict how frequently a particular worker can work on a particular task.', self.is_sandbox)
        assert block_qual_id is not None, 'Hits could not be created as block qualification could not be acquired. Shutting down server.'
        qualifications.append({'QualificationTypeId': block_qual_id, 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'})
    if self.has_time_limit:
        block_qual_name = '{}-max-daily-time'.format(self.task_group_id)
        if self.opt['max_time_qual'] is not None:
            block_qual_name = self.opt['max_time_qual']
        self.max_time_qual = block_qual_name
        block_qual_id = mturk_utils.find_or_create_qualification(block_qual_name, 'A soft ban from working on this HIT or HITs by this requester based on a maximum amount of daily work time set by the requester.', self.is_sandbox)
        assert block_qual_id is not None, 'Hits could not be created as a time block qualification could not be acquired. Shutting down server.'
        qualifications.append({'QualificationTypeId': block_qual_id, 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'})
    if self.is_unique or self.max_hits_per_worker > 0:
        self.unique_qual_name = self.opt.get('unique_qual_name')
        if self.unique_qual_name is None:
            self.unique_qual_name = self.task_group_id + '_max_submissions'
        self.unique_qual_id = mturk_utils.find_or_create_qualification(self.unique_qual_name, 'Prevents workers from completing a task too frequently', self.is_sandbox)
        qualifications.append({'QualificationTypeId': self.unique_qual_id, 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'})
    self.qualifications = qualifications
    return qualifications.copy()