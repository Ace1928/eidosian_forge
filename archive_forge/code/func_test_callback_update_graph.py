import unittest
import subprocess
import time
from gensim.models import LdaModel
from gensim.test.utils import datapath, common_dictionary
from gensim.corpora import MmCorpus
from gensim.models.callbacks import CoherenceMetric
def test_callback_update_graph(self):
    with subprocess.Popen(['python', '-m', 'visdom.server', '-port', str(self.port)]) as proc:
        viz = Visdom(server=self.host, port=self.port)
        for attempt in range(5):
            time.sleep(1.0)
            if viz.check_connection():
                break
        assert viz.check_connection()
        viz.close()
        self.model.update(self.corpus)
        proc.kill()