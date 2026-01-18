import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.skipif(noboto3 or not fakes3, reason='boto3 or fakes3 library is not available')
def test_aws_keys_from_env():
    """
    Function to ensure the DataSink can successfully read in AWS
    credentials from the environment variables
    """
    ds = nio.DataSink()
    aws_access_key_id = 'ABCDACCESS'
    aws_secret_access_key = 'DEFGSECRET'
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    access_key_test, secret_key_test = ds._return_aws_keys()
    assert aws_access_key_id == access_key_test
    assert aws_secret_access_key == secret_key_test