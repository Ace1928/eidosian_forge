import logging
def get_global_history_recorder():
    global HISTORY_RECORDER
    if HISTORY_RECORDER is None:
        HISTORY_RECORDER = HistoryRecorder()
    return HISTORY_RECORDER