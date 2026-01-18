import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def best_score_cb(result):
    global best_score
    best_score = result.best['score']