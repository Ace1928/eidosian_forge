from __future__ import absolute_import
import os
from os import path
import sys
import uuid
import logging
from kubernetes import client, config
from . import tracker
import yaml
def create_job_manifest(envs, commands, name, image, template_file):
    if template_file is not None:
        with open(template_file) as f:
            job = yaml.safe_load(f)
            job['metadata']['name'] = name
            job['spec']['template']['metadata']['labels']['app'] = name
            job['spec']['template']['spec']['containers'][0]['image'] = image
            job['spec']['template']['spec']['containers'][0]['command'] = commands
            job['spec']['template']['spec']['containers'][0]['name'] = name
            job['spec']['template']['spec']['containers'][0]['env'] = envs
            job['spec']['template']['spec']['containers'][0]['command'] = commands
    else:
        container = client.V1Container(image=image, command=commands, name=name, env=envs)
        pod_temp = client.V1PodTemplateSpec(spec=client.V1PodSpec(restart_policy='OnFailure', containers=[container]), metadata=client.V1ObjectMeta(name=name, labels={'app': name}))
        job = client.V1Job(api_version='batch/v1', kind='Job', spec=client.V1JobSpec(template=pod_temp), metadata=client.V1ObjectMeta(name=name))
    return job