import re
import torch
def identify_failure_mode(self, text):
    if re.search(self.failure_regexes[ISAID], text, flags=re.I):
        return ISAID
    elif re.search(self.failure_regexes[NOTSENSE], text, flags=re.I):
        return NOTSENSE
    elif re.search(self.failure_regexes[UM], text, flags=re.I):
        return UM
    elif re.search(self.failure_regexes[YOUWHAT], text, flags=re.I):
        return YOUWHAT
    elif re.search(self.failure_regexes[WHATYOU], text, flags=re.I):
        return WHATYOU
    elif re.search(self.failure_regexes[WHATDO], text, flags=re.I):
        return WHATDO
    else:
        return None