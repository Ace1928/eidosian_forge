import os
import time
def get_eta(start_time, current, total, enough_samples=3, last_updates=None, n_recent=10):
    if start_time is None:
        return None
    if not total:
        return None
    if current < enough_samples:
        return None
    if current > total:
        return None
    elapsed = time.time() - start_time
    if elapsed < 2.0:
        return None
    total_duration = float(elapsed) * float(total) / float(current)
    if last_updates and len(last_updates) >= n_recent:
        avg = sum(last_updates) / float(len(last_updates))
        time_left = avg * (total - current)
        old_time_left = total_duration - elapsed
        return (time_left + old_time_left) / 2
    return total_duration - elapsed