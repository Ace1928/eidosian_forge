def del_attr(self):
    return delattr(getattr(self, target), attr)