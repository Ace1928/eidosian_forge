import logging
def get_eval_logger():
    if eval_logger is None:
        init()
    return eval_logger