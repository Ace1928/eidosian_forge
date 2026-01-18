from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_complicated_nested_rest(self):

    class BarsController(RestController):
        data = [['zero-zero', 'zero-one'], ['one-zero', 'one-one']]

        @expose()
        def get_one(self, foo_id, id):
            return self.data[int(foo_id)][int(id)]

        @expose('json')
        def get_all(self, foo_id):
            return dict(items=self.data[int(foo_id)])

        @expose()
        def new(self, foo_id):
            return 'NEW FOR %s' % foo_id

        @expose()
        def post(self, foo_id, value):
            foo_id = int(foo_id)
            if len(self.data) < foo_id + 1:
                self.data.extend([[]] * (foo_id - len(self.data) + 1))
            self.data[foo_id].append(value)
            response.status = 302
            return 'CREATED FOR %s' % foo_id

        @expose()
        def edit(self, foo_id, id):
            return 'EDIT %s' % self.data[int(foo_id)][int(id)]

        @expose()
        def put(self, foo_id, id, value):
            self.data[int(foo_id)][int(id)] = value
            return 'UPDATED'

        @expose()
        def get_delete(self, foo_id, id):
            return 'DELETE %s' % self.data[int(foo_id)][int(id)]

        @expose()
        def delete(self, foo_id, id):
            del self.data[int(foo_id)][int(id)]
            return 'DELETED'

    class FoosController(RestController):
        data = ['zero', 'one']
        bars = BarsController()

        @expose()
        def get_one(self, id):
            return self.data[int(id)]

        @expose('json')
        def get_all(self):
            return dict(items=self.data)

        @expose()
        def new(self):
            return 'NEW'

        @expose()
        def edit(self, id):
            return 'EDIT %s' % self.data[int(id)]

        @expose()
        def post(self, value):
            self.data.append(value)
            response.status = 302
            return 'CREATED'

        @expose()
        def put(self, id, value):
            self.data[int(id)] = value
            return 'UPDATED'

        @expose()
        def get_delete(self, id):
            return 'DELETE %s' % self.data[int(id)]

        @expose()
        def delete(self, id):
            del self.data[int(id)]
            return 'DELETED'

    class RootController(object):
        foos = FoosController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foos')
    assert r.status_int == 200
    assert r.body == dumps(dict(items=FoosController.data)).encode('utf-8')
    r = app.get('/foos/1/bars')
    assert r.status_int == 200
    assert r.body == dumps(dict(items=BarsController.data[1])).encode('utf-8')
    for i, value in enumerate(FoosController.data):
        r = app.get('/foos/%d' % i)
        assert r.status_int == 200
        assert r.body == value.encode('utf-8')
    for i, value in enumerate(FoosController.data):
        for j, value in enumerate(BarsController.data[i]):
            r = app.get('/foos/%s/bars/%s' % (i, j))
            assert r.status_int == 200
            assert r.body == value.encode('utf-8')
    r = app.post('/foos', {'value': 'two'})
    assert r.status_int == 302
    assert r.body == b'CREATED'
    r = app.get('/foos/2')
    assert r.status_int == 200
    assert r.body == b'two'
    r = app.post('/foos/2/bars', {'value': 'two-zero'})
    assert r.status_int == 302
    assert r.body == b'CREATED FOR 2'
    r = app.get('/foos/2/bars/0')
    assert r.status_int == 200
    assert r.body == b'two-zero'
    r = app.get('/foos/1/edit')
    assert r.status_int == 200
    assert r.body == b'EDIT one'
    r = app.get('/foos/1/bars/1/edit')
    assert r.status_int == 200
    assert r.body == b'EDIT one-one'
    r = app.put('/foos/2', {'value': 'TWO'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/foos/2')
    assert r.status_int == 200
    assert r.body == b'TWO'
    r = app.put('/foos/2/bars/0', {'value': 'TWO-ZERO'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/foos/2/bars/0')
    assert r.status_int == 200
    assert r.body == b'TWO-ZERO'
    r = app.get('/foos/2?_method=put', {'value': 'TWO!'}, status=405)
    assert r.status_int == 405
    r = app.get('/foos/2')
    assert r.status_int == 200
    assert r.body == b'TWO'
    r = app.get('/foos/2/bars/0?_method=put', {'value': 'ZERO-TWO!'}, status=405)
    assert r.status_int == 405
    r = app.get('/foos/2/bars/0')
    assert r.status_int == 200
    assert r.body == b'TWO-ZERO'
    r = app.post('/foos/2?_method=put', {'value': 'TWO!'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/foos/2')
    assert r.status_int == 200
    assert r.body == b'TWO!'
    r = app.post('/foos/2/bars/0?_method=put', {'value': 'TWO-ZERO!'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/foos/2/bars/0')
    assert r.status_int == 200
    assert r.body == b'TWO-ZERO!'
    r = app.get('/foos/2/delete')
    assert r.status_int == 200
    assert r.body == b'DELETE TWO!'
    r = app.get('/foos/2/bars/0/delete')
    assert r.status_int == 200
    assert r.body == b'DELETE TWO-ZERO!'
    r = app.delete('/foos/2/bars/0')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/foos/2/bars')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 0
    r = app.delete('/foos/2')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/foos')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 2
    r = app.get('/foos/1/bars/1?_method=DELETE', status=405)
    assert r.status_int == 405
    r = app.get('/foos/1/bars')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 2
    r = app.get('/foos/1?_method=DELETE', status=405)
    assert r.status_int == 405
    r = app.get('/foos')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 2
    r = app.post('/foos/1/bars/1?_method=DELETE')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/foos/1/bars')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 1
    r = app.post('/foos/1?_method=DELETE')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/foos')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 1