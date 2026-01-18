from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler
def get_error_handler():
    return ErrorHandler()