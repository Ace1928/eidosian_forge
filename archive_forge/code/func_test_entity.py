from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.geometry import (Circle, Ellipse, Point, Line, Parabola,
from sympy.geometry.entity import scale, GeometryEntity
from sympy.testing.pytest import raises
def test_entity():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert GeometryEntity(x, y) in GeometryEntity(x, y)
    raises(NotImplementedError, lambda: Point(0, 0) in GeometryEntity(x, y))
    assert GeometryEntity(x, y) == GeometryEntity(x, y)
    assert GeometryEntity(x, y).equals(GeometryEntity(x, y))
    c = Circle((0, 0), 5)
    assert GeometryEntity.encloses(c, Point(0, 0))
    assert GeometryEntity.encloses(c, Segment((0, 0), (1, 1)))
    assert GeometryEntity.encloses(c, Line((0, 0), (1, 1))) is False
    assert GeometryEntity.encloses(c, Circle((0, 0), 4))
    assert GeometryEntity.encloses(c, Polygon(Point(0, 0), Point(1, 0), Point(0, 1)))
    assert GeometryEntity.encloses(c, RegularPolygon(Point(8, 8), 1, 3)) is False