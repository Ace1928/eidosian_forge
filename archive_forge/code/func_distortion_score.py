import warnings
from collections import defaultdict
from math import log
def distortion_score(self, hypothesis, next_src_phrase_span):
    if not hypothesis.src_phrase_span:
        return 0.0
    next_src_phrase_start = next_src_phrase_span[0]
    prev_src_phrase_end = hypothesis.src_phrase_span[1]
    distortion_distance = next_src_phrase_start - prev_src_phrase_end
    return abs(distortion_distance) * self.__log_distortion_factor