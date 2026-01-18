import copy
import datasets
import itertools
def get_threads_from_root(root_id):
    all_threads = []
    thread = [messages[root_id]]
    for cid in nodes[root_id]:
        all_threads += follow(thread, cid)
    return all_threads