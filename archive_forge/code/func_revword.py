import random
def revword(word):
    if random.randint(1, 2) == 1:
        return word[::-1]
    return word