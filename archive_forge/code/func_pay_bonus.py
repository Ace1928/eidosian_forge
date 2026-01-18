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
def pay_bonus(self, worker_id, bonus_amount, assignment_id, reason, unique_request_token):
    """
        Handles paying bonus to a turker, fails for insufficient funds.

        Returns True on success and False on failure
        """
    total_cost = mturk_utils.calculate_mturk_cost(payment_opt={'type': 'bonus', 'amount': bonus_amount})
    if not mturk_utils.check_mturk_balance(balance_needed=total_cost, is_sandbox=self.is_sandbox):
        shared_utils.print_and_log(logging.WARN, 'Cannot pay bonus. Reason: Insufficient funds in your MTurk account.', should_print=True)
        return False
    client = mturk_utils.get_mturk_client(self.is_sandbox)
    client.send_bonus(WorkerId=worker_id, BonusAmount=str(bonus_amount), AssignmentId=assignment_id, Reason=reason, UniqueRequestToken=unique_request_token)
    if self.db_logger is not None:
        self.db_logger.log_pay_extra_bonus(worker_id, assignment_id, bonus_amount, reason)
    shared_utils.print_and_log(logging.INFO, 'Paid ${} bonus to WorkerId: {}'.format(bonus_amount, worker_id))
    return True