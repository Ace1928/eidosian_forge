from __future__ import absolute_import
import os
from os import path
import sys
import uuid
import logging
from kubernetes import client, config
from . import tracker
import yaml
def create_wk_manifest(wk_id, wk_num, ps_num, job_name, envs, image, commands, template_file):
    envs.append(client.V1EnvVar(name='DMLC_WORKER_ID', value=wk_id))
    envs.append(client.V1EnvVar(name='DMLC_SERVER_ID', value='0'))
    envs.append(client.V1EnvVar(name='DMLC_ROLE', value='worker'))
    if job_name is not None:
        name = 'mx-' + job_name + '-worker-' + wk_id
    else:
        name = 'mx-worker-' + wk_id
    return create_job_manifest(envs, commands, name, image, template_file)