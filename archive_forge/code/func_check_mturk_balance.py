import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def check_mturk_balance(balance_needed, is_sandbox):
    """
    Checks to see if there is at least balance_needed amount in the requester account,
    returns True if the balance is greater than balance_needed.
    """
    client = get_mturk_client(is_sandbox)
    try:
        user_balance = float(client.get_account_balance()['AvailableBalance'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'RequestError':
            print('ERROR: To use the MTurk API, you will need an Amazon Web Services (AWS) Account. Your AWS account must be linked to your Amazon Mechanical Turk Account. Visit https://requestersandbox.mturk.com/developer to get started. (Note: if you have recently linked your account, please wait for a couple minutes before trying again.)\n')
            quit()
        else:
            raise
    if user_balance < balance_needed:
        print('You might not have enough money in your MTurk account. Please go to https://requester.mturk.com/account and increase your balance to at least ${}, and then try again.'.format(balance_needed))
        return False
    else:
        return True