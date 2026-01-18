import unittest
import logging
import time
def hashfunc(msg):
    return hmac.new('mysecret', msg)