import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def setup_aws_credentials():
    try:
        boto3.Session(profile_name=aws_profile_name)
    except ProfileNotFound:
        print('AWS credentials not found. Please create an IAM user with programmatic access and AdministratorAccess policy at https://console.aws.amazon.com/iam/ (On the "Set permissions" page, choose "Attach existing policies directly" and then select "AdministratorAccess" policy). After creating the IAM user, please enter the user\'s Access Key ID and Secret Access Key below:')
        aws_access_key_id = input('Access Key ID: ')
        aws_secret_access_key = input('Secret Access Key: ')
        if not os.path.exists(os.path.expanduser('~/.aws/')):
            os.makedirs(os.path.expanduser('~/.aws/'))
        aws_credentials_file_path = '~/.aws/credentials'
        aws_credentials_file_string = None
        expanded_aws_file_path = os.path.expanduser(aws_credentials_file_path)
        if os.path.exists(expanded_aws_file_path):
            with open(expanded_aws_file_path, 'r') as aws_credentials_file:
                aws_credentials_file_string = aws_credentials_file.read()
        with open(expanded_aws_file_path, 'a+') as aws_credentials_file:
            if aws_credentials_file_string:
                if aws_credentials_file_string.endswith('\n\n'):
                    pass
                elif aws_credentials_file_string.endswith('\n'):
                    aws_credentials_file.write('\n')
                else:
                    aws_credentials_file.write('\n\n')
            aws_credentials_file.write('[{}]\n'.format(aws_profile_name))
            aws_credentials_file.write('aws_access_key_id={}\n'.format(aws_access_key_id))
            aws_credentials_file.write('aws_secret_access_key={}\n'.format(aws_secret_access_key))
        print('AWS credentials successfully saved in {} file.\n'.format(aws_credentials_file_path))
    os.environ['AWS_PROFILE'] = aws_profile_name