import yaml
import os
def read_github_workflow_file():
    with open(github_workflow_file, 'r') as gh_file:
        try:
            return yaml.safe_load(gh_file)
        except yaml.YAMLError as exc:
            print(exc)