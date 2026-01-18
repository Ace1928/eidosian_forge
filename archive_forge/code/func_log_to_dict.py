import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def log_to_dict(logfile):
    """
    Function to extract log node dictionaries into a list of python
    dictionaries and return the list as well as the final node

    Parameters
    ----------
    logfile : string
        path to the json-formatted log file generated from a nipype
        workflow execution

    Returns
    -------
    nodes_list : list
        a list of python dictionaries containing the runtime info
        for each nipype node
    """
    with open(logfile, 'r') as content:
        lines = content.readlines()
    nodes_list = [json.loads(l) for l in lines]
    return nodes_list