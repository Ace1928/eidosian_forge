import os
import time
def print_message(message: str) -> None:
    time_now = time.time()
    print('WANDB_STARTUP_DEBUG', time_now, message)