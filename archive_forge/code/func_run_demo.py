import statemachine
import documentsignoffstate
def run_demo():
    import random
    doc = Document()
    print(doc)
    doc.create()
    print(doc)
    print(doc.state.description)
    while not isinstance(doc._state, documentsignoffstate.Approved):
        print('...submit')
        doc.submit()
        print(doc)
        print(doc.state.description)
        if random.randint(1, 10) > 3:
            print('...reject')
            doc.reject()
        else:
            print('...approve')
            doc.approve()
        print(doc)
        print(doc.state.description)
    doc.activate()
    print(doc)
    print(doc.state.description)