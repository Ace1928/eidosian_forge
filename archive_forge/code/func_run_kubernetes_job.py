import logging
import os
import time
from datetime import datetime
from shlex import quote, split
from threading import RLock
import kubernetes
from kubernetes.config.config_exception import ConfigException
import docker
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
def run_kubernetes_job(project_name, active_run, image_tag, image_digest, command, env_vars, kube_context=None, job_template=None):
    job_template = _get_kubernetes_job_definition(project_name, image_tag, image_digest, _get_run_command(command), env_vars, job_template)
    job_name = job_template['metadata']['name']
    job_namespace = job_template['metadata']['namespace']
    _load_kube_context(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_instance.create_namespaced_job(namespace=job_namespace, body=job_template, pretty=True)
    return KubernetesSubmittedRun(active_run.info.run_id, job_name, job_namespace)