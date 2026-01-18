from __future__ import absolute_import
import os
from os import path
import sys
import uuid
import logging
from kubernetes import client, config
from . import tracker
import yaml
def kubernetes_submit(nworker, nserver, pass_envs):
    sv_image = args.kube_server_image
    wk_image = args.kube_worker_image
    if args.jobname is not None:
        r_uri = 'mx-' + args.jobname + '-sched'
    else:
        r_uri = 'mx-sched'
    r_port = 9091
    sd_envs = create_env(r_uri, r_port, nserver, nworker)
    mn_jobs = []
    mn_sh_job = create_sched_job_manifest(str(nworker), str(nserver), sd_envs, sv_image, args.command)
    mn_sh_svc = create_sched_svc_manifest(r_uri, r_port)
    for i in range(nserver):
        envs = create_env(r_uri, r_port, nserver, nworker)
        mn_sv = create_ps_manifest(str(i), str(nserver), args.jobname, envs, sv_image, args.command, args.kube_server_template)
        mn_jobs.append(mn_sv)
    for i in range(nworker):
        envs = create_env(r_uri, r_port, nserver, nworker)
        mn_wk = create_wk_manifest(str(i), str(nworker), str(nserver), args.jobname, envs, wk_image, args.command, args.kube_worker_template)
        mn_jobs.append(mn_wk)
    config.load_kube_config()
    k8s_coreapi = client.CoreV1Api()
    k8s_batch = client.BatchV1Api()
    resp = k8s_batch.create_namespaced_job(namespace=args.kube_namespace, body=mn_sh_job)
    print(resp.kind + ' ' + resp.metadata.name + ' is created.')
    resp = k8s_coreapi.create_namespaced_service(namespace='default', body=mn_sh_svc)
    print(resp.kind + ' ' + resp.metadata.name + ' is created.')
    for m in mn_jobs:
        resp = k8s_batch.create_namespaced_job(body=m, namespace='default')
        print(resp.kind + ' ' + resp.metadata.name + ' is created.')
    return kubernetes_submit