import unittest
import pickle
from .common import MTurkCommon
def create_hit_result(self):
    return self.conn.create_hit(question=self.get_question(), **self.get_hit_params())