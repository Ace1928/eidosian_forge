import re
def scan_once(string, idx):
    try:
        return _scan_once(string, idx)
    finally:
        memo.clear()