import dill
def get_fun_with_strftime():

    def fun_with_strftime():
        import datetime
        return datetime.datetime.strptime('04-01-1943', '%d-%m-%Y').strftime('%Y-%m-%d %H:%M:%S')
    return fun_with_strftime