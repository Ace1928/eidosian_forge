import time
from argparse import ArgumentParser
from typing import Optional
from huggingface_hub import HfApi, create_branch, get_repo_discussions
from datasets import get_dataset_config_names, get_dataset_default_config_name, load_dataset
from datasets.commands import BaseDatasetsCLICommand
def delete_files(dataset_id, revision=None, token=None):
    dataset_name = dataset_id.split('/')[-1]
    hf_api = HfApi(token=token)
    repo_files = hf_api.list_repo_files(dataset_id, repo_type='dataset')
    if repo_files:
        legacy_json_file = []
        python_files = []
        data_files = []
        for filename in repo_files:
            if filename in {'.gitattributes', 'README.md'}:
                continue
            elif filename == f'{dataset_name}.py':
                hf_api.delete_file(filename, dataset_id, repo_type='dataset', revision=revision, commit_message='Delete loading script')
            elif filename == 'dataset_infos.json':
                legacy_json_file.append(filename)
            elif filename.endswith('.py'):
                python_files.append(filename)
            else:
                data_files.append(filename)
        if legacy_json_file:
            hf_api.delete_file('dataset_infos.json', dataset_id, repo_type='dataset', revision=revision, commit_message='Delete legacy dataset_infos.json')
        if python_files:
            for filename in python_files:
                hf_api.delete_file(filename, dataset_id, repo_type='dataset', revision=revision, commit_message='Delete loading script auxiliary file')
        if data_files:
            for filename in data_files:
                hf_api.delete_file(filename, dataset_id, repo_type='dataset', revision=revision, commit_message='Delete data file')