import time
import pickle
def save_log_entries_as_pickle(self):
    with open(self.outfile, 'wb') as f:
        pickle.dump(self.log_entries, f)