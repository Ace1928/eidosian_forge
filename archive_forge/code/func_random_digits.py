import random
import string
def random_digits(n=8):
    return ''.join([random.choice(string.digits) for _ in range(n)])