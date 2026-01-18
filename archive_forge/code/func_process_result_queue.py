import logging
import queue
from multiprocessing import Pool, Queue, cpu_count
import numpy as np
from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState
def process_result_queue(force=False):
    """
            Clear the result queue, merging all intermediate results, and update the
            LDA model if necessary.

            """
    merged_new = False
    while not result_queue.empty():
        other.merge(result_queue.get())
        queue_size[0] -= 1
        merged_new = True
    if force and merged_new and (queue_size[0] == 0) or other.numdocs >= updateafter:
        self.do_mstep(rho(), other, pass_ > 0)
        other.reset()
        if eval_every > 0 and (force or self.num_updates / updateafter % eval_every == 0):
            self.log_perplexity(chunk, total_docs=lencorpus)