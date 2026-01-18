import argparse
import json
import sys
import pkg_resources
import pbr.version
def get_sha(args):
    sha = _get_info(args.name)['sha']
    if sha:
        print(sha)