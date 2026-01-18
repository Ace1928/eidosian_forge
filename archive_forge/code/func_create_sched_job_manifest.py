from __future__ import absolute_import
import os
from os import path
import sys
import uuid
import logging
from kubernetes import client, config
from . import tracker
import yaml
def create_sched_job_manifest(wk_num, ps_num, envs, image, commands):
    envs.append(client.V1EnvVar(name='DMLC_ROLE', value='scheduler'))
    name = ''
    for i in envs:
        if i.name == 'DMLC_PS_ROOT_URI':
            name = i.value
            break
    return create_job_manifest(envs, commands, name, image, None)