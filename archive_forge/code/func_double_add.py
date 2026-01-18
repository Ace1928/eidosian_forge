import dill
@quad(a=0, b=2)
def double_add(*args):
    return sum(args)