from sympy.core import expand
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sec
from sympy.geometry.line import Segment2D
from sympy.geometry.point import Point2D
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
from sympy.testing.pytest import raises, slow
from sympy.integrals.integrals import integrate
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.functions.elementary.miscellaneous import Max
@slow
def test_ellipse_geom():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    y1 = Symbol('y1', real=True)
    half = S.Half
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p4 = Point(0, 1)
    e1 = Ellipse(p1, 1, 1)
    e2 = Ellipse(p2, half, 1)
    e3 = Ellipse(p1, y1, y1)
    c1 = Circle(p1, 1)
    c2 = Circle(p2, 1)
    c3 = Circle(Point(sqrt(2), sqrt(2)), 1)
    l1 = Line(p1, p2)
    cen, rad = (Point(3 * half, 2), 5 * half)
    assert Circle(Point(0, 0), Point(3, 0), Point(0, 4)) == Circle(cen, rad)
    assert Circle(Point(0, 0), Point(1, 1), Point(2, 2)) == Segment2D(Point2D(0, 0), Point2D(2, 2))
    raises(ValueError, lambda: Ellipse(None, None, None, 1))
    raises(ValueError, lambda: Ellipse())
    raises(GeometryError, lambda: Circle(Point(0, 0)))
    raises(GeometryError, lambda: Circle(Symbol('x') * Symbol('y')))
    assert Ellipse(None, 1, 1).center == Point(0, 0)
    assert e1 == c1
    assert e1 != e2
    assert e1 != l1
    assert p4 in e1
    assert e1 in e1
    assert e2 in e2
    assert 1 not in e2
    assert p2 not in e2
    assert e1.area == pi
    assert e2.area == pi / 2
    assert e3.area == pi * y1 * abs(y1)
    assert c1.area == e1.area
    assert c1.circumference == e1.circumference
    assert e3.circumference == 2 * pi * y1
    assert e1.plot_interval() == e2.plot_interval() == [t, -pi, pi]
    assert e1.plot_interval(x) == e2.plot_interval(x) == [x, -pi, pi]
    assert c1.minor == 1
    assert c1.major == 1
    assert c1.hradius == 1
    assert c1.vradius == 1
    assert Ellipse((1, 1), 0, 0) == Point(1, 1)
    assert Ellipse((1, 1), 1, 0) == Segment(Point(0, 1), Point(2, 1))
    assert Ellipse((1, 1), 0, 1) == Segment(Point(1, 0), Point(1, 2))
    assert hash(c1) == hash(Circle(Point(1, 0), Point(0, 1), Point(0, -1)))
    assert c1 in e1
    assert (Line(p1, p2) in e1) is False
    assert e1.__cmp__(e1) == 0
    assert e1.__cmp__(Point(0, 0)) > 0
    assert e1.encloses(Segment(Point(-0.5, -0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(Line(p1, p2)) is False
    assert e1.encloses(Ray(p1, p2)) is False
    assert e1.encloses(e1) is False
    assert e1.encloses(Polygon(Point(-0.5, -0.5), Point(-0.5, 0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(RegularPolygon(p1, 0.5, 3)) is True
    assert e1.encloses(RegularPolygon(p1, 5, 3)) is False
    assert e1.encloses(RegularPolygon(p2, 5, 3)) is False
    assert e2.arbitrary_point() in e2
    raises(ValueError, lambda: Ellipse(Point(x, y), 1, 1).arbitrary_point(parameter='x'))
    f1, f2 = (Point(sqrt(12), 0), Point(-sqrt(12), 0))
    ef = Ellipse(Point(0, 0), 4, 2)
    assert ef.foci in [(f1, f2), (f2, f1)]
    v = sqrt(2) / 2
    p1_1 = Point(v, v)
    p1_2 = p2 + Point(half, 0)
    p1_3 = p2 + Point(0, 1)
    assert e1.tangent_lines(p4) == c1.tangent_lines(p4)
    assert e2.tangent_lines(p1_2) == [Line(Point(Rational(3, 2), 1), Point(Rational(3, 2), S.Half))]
    assert e2.tangent_lines(p1_3) == [Line(Point(1, 2), Point(Rational(5, 4), 2))]
    assert c1.tangent_lines(p1_1) != [Line(p1_1, Point(0, sqrt(2)))]
    assert c1.tangent_lines(p1) == []
    assert e2.is_tangent(Line(p1_2, p2 + Point(half, 1)))
    assert e2.is_tangent(Line(p1_3, p2 + Point(half, 1)))
    assert c1.is_tangent(Line(p1_1, Point(0, sqrt(2))))
    assert e1.is_tangent(Line(Point(0, 0), Point(1, 1))) is False
    assert c1.is_tangent(e1) is True
    assert c1.is_tangent(Ellipse(Point(2, 0), 1, 1)) is True
    assert c1.is_tangent(Polygon(Point(1, 1), Point(1, -1), Point(2, 0))) is True
    assert c1.is_tangent(Polygon(Point(1, 1), Point(1, 0), Point(2, 0))) is False
    assert Circle(Point(5, 5), 3).is_tangent(Circle(Point(0, 5), 1)) is False
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(0, 0)) == [Line(Point(0, 0), Point(Rational(77, 25), Rational(132, 25))), Line(Point(0, 0), Point(Rational(33, 5), Rational(22, 5)))]
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(3, 4)) == [Line(Point(3, 4), Point(4, 4)), Line(Point(3, 4), Point(3, 5))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(3, 3)) == [Line(Point(3, 3), Point(4, 3)), Line(Point(3, 3), Point(3, 4))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(5 - 2 * sqrt(2), 5)) == [Line(Point(5 - 2 * sqrt(2), 5), Point(5 - sqrt(2), 5 - sqrt(2))), Line(Point(5 - 2 * sqrt(2), 5), Point(5 - sqrt(2), 5 + sqrt(2)))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(4, 0)) == [Line(Point(4, 0), Point(Rational(40, 13), Rational(5, 13))), Line(Point(4, 0), Point(5, 0))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(0, 6)) == [Line(Point(0, 6), Point(0, 7)), Line(Point(0, 6), Point(Rational(5, 13), Rational(90, 13)))]

    def lines_close(l1, l2, prec):
        """ tests whether l1 and 12 are within 10**(-prec)
        of each other """
        return abs(l1.p1 - l2.p1) < 10 ** (-prec) and abs(l1.p2 - l2.p2) < 10 ** (-prec)

    def line_list_close(ll1, ll2, prec):
        return all((lines_close(l1, l2, prec) for l1, l2 in zip(ll1, ll2)))
    e = Ellipse(Point(0, 0), 2, 1)
    assert e.normal_lines(Point(0, 0)) == [Line(Point(0, 0), Point(0, 1)), Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines(Point(1, 0)) == [Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines((0, 1)) == [Line(Point(0, 0), Point(0, 1))]
    assert line_list_close(e.normal_lines(Point(1, 1), 2), [Line(Point(Rational(-51, 26), Rational(-1, 5)), Point(Rational(-25, 26), Rational(17, 83))), Line(Point(Rational(28, 29), Rational(-7, 8)), Point(Rational(57, 29), Rational(-9, 2)))], 2)
    p = Point(sqrt(3), S.Half)
    assert p in e
    assert line_list_close(e.normal_lines(p, 2), [Line(Point(Rational(-341, 171), Rational(-1, 13)), Point(Rational(-170, 171), Rational(5, 64))), Line(Point(Rational(26, 15), Rational(-1, 2)), Point(Rational(41, 15), Rational(-43, 26)))], 2)
    e = Ellipse((0, 0), 2, 2 * sqrt(3) / 3)
    assert line_list_close(e.normal_lines((1, 1), 2), [Line(Point(Rational(-64, 33), Rational(-20, 71)), Point(Rational(-31, 33), Rational(2, 13))), Line(Point(1, -1), Point(2, -4))], 2)
    e = Ellipse((0, 0), x, 1)
    assert e.normal_lines((x + 1, 0)) == [Line(Point(0, 0), Point(1, 0))]
    raises(NotImplementedError, lambda: e.normal_lines((x + 1, 1)))
    major = 3
    minor = 1
    e4 = Ellipse(p2, minor, major)
    assert e4.focus_distance == sqrt(major ** 2 - minor ** 2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major * (1 - ecc)
    assert e4.apoapsis == major * (1 + ecc)
    assert e4.semilatus_rectum == major * (1 - ecc ** 2)
    e4 = Ellipse(p2, major, minor)
    assert e4.focus_distance == sqrt(major ** 2 - minor ** 2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major * (1 - ecc)
    assert e4.apoapsis == major * (1 + ecc)
    l1 = Line(Point(1, -5), Point(1, 5))
    l2 = Line(Point(-5, -1), Point(5, -1))
    l3 = Line(Point(-1, -1), Point(1, 1))
    l4 = Line(Point(-10, 0), Point(0, 10))
    pts_c1_l3 = [Point(sqrt(2) / 2, sqrt(2) / 2), Point(-sqrt(2) / 2, -sqrt(2) / 2)]
    assert intersection(e2, l4) == []
    assert intersection(c1, Point(1, 0)) == [Point(1, 0)]
    assert intersection(c1, l1) == [Point(1, 0)]
    assert intersection(c1, l2) == [Point(0, -1)]
    assert intersection(c1, l3) in [pts_c1_l3, [pts_c1_l3[1], pts_c1_l3[0]]]
    assert intersection(c1, c2) == [Point(0, 1), Point(1, 0)]
    assert intersection(c1, c3) == [Point(sqrt(2) / 2, sqrt(2) / 2)]
    assert e1.intersection(l1) == [Point(1, 0)]
    assert e2.intersection(l4) == []
    assert e1.intersection(Circle(Point(0, 2), 1)) == [Point(0, 1)]
    assert e1.intersection(Circle(Point(5, 0), 1)) == []
    assert e1.intersection(Ellipse(Point(2, 0), 1, 1)) == [Point(1, 0)]
    assert e1.intersection(Ellipse(Point(5, 0), 1, 1)) == []
    assert e1.intersection(Point(2, 0)) == []
    assert e1.intersection(e1) == e1
    assert intersection(Ellipse(Point(0, 0), 2, 1), Ellipse(Point(3, 0), 1, 2)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(3, 0), 1)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(7, 0), 1)) == []
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 1, 0.2)) == [Point(5, 0)]
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 0.999, 0.2)) == []
    assert Circle((0, 0), S.Half).intersection(Triangle((-1, 0), (1, 0), (0, 1))) == [Point(Rational(-1, 2), 0), Point(S.Half, 0)]
    raises(TypeError, lambda: intersection(e2, Line((0, 0, 0), (0, 0, 1))))
    raises(TypeError, lambda: intersection(e2, Rational(12)))
    raises(TypeError, lambda: Ellipse.intersection(e2, 1))
    csmall = Circle(p1, 3)
    cbig = Circle(p1, 5)
    cout = Circle(Point(5, 5), 1)
    assert csmall.intersection(cbig) == []
    assert csmall.intersection(cout) == []
    assert csmall.intersection(csmall) == csmall
    v = sqrt(2)
    t1 = Triangle(Point(0, v), Point(0, -v), Point(v, 0))
    points = intersection(t1, c1)
    assert len(points) == 4
    assert Point(0, 1) in points
    assert Point(0, -1) in points
    assert Point(v / 2, v / 2) in points
    assert Point(v / 2, -v / 2) in points
    circ = Circle(Point(0, 0), 5)
    elip = Ellipse(Point(0, 0), 5, 20)
    assert intersection(circ, elip) in [[Point(5, 0), Point(-5, 0)], [Point(-5, 0), Point(5, 0)]]
    assert elip.tangent_lines(Point(0, 0)) == []
    elip = Ellipse(Point(0, 0), 3, 2)
    assert elip.tangent_lines(Point(3, 0)) == [Line(Point(3, 0), Point(3, -12))]
    e1 = Ellipse(Point(0, 0), 5, 10)
    e2 = Ellipse(Point(2, 1), 4, 8)
    a = Rational(53, 17)
    c = 2 * sqrt(3991) / 17
    ans = [Point(a - c / 8, a / 2 + c), Point(a + c / 8, a / 2 - c)]
    assert e1.intersection(e2) == ans
    e2 = Ellipse(Point(x, y), 4, 8)
    c = sqrt(3991)
    ans = [Point(-c / 68 + a, c * Rational(2, 17) + a / 2), Point(c / 68 + a, c * Rational(-2, 17) + a / 2)]
    assert [p.subs({x: 2, y: 1}) for p in e1.intersection(e2)] == ans
    assert e3.is_tangent(e3.tangent_lines(p1 + Point(y1, 0))[0])
    e = Ellipse((1, 2), 3, 2)
    assert e.tangent_lines(Point(10, 0)) == [Line(Point(10, 0), Point(1, 0)), Line(Point(10, 0), Point(Rational(14, 5), Rational(18, 5)))]
    e = Ellipse((0, 0), 1, 2)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    e = Ellipse((0, 0), 2, 1)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    assert c1.encloses_point(Point(1, 0)) is False
    assert c1.encloses_point(Point(0.3, 0.4)) is True
    assert e.scale(2, 3) == Ellipse((0, 0), 4, 3)
    assert e.scale(3, 6) == Ellipse((0, 0), 6, 6)
    assert e.rotate(pi) == e
    assert e.rotate(pi, (1, 2)) == Ellipse(Point(2, 4), 2, 1)
    raises(NotImplementedError, lambda: e.rotate(pi / 3))
    cir = Circle(Point(1, 0), 1)
    assert cir.rotate(pi / 2) == Circle(Point(0, 1), 1)
    assert cir.rotate(pi / 3) == Circle(Point(S.Half, sqrt(3) / 2), 1)
    assert cir.rotate(pi / 3, Point(1, 0)) == Circle(Point(1, 0), 1)
    assert cir.rotate(pi / 3, Point(0, 1)) == Circle(Point(S.Half + sqrt(3) / 2, S.Half + sqrt(3) / 2), 1)