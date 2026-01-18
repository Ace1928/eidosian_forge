from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3, DirectionalLight
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import MouseWatcher, ModifierButtons, PGMouseWatcherBackground
from panda3d.core import WindowProperties
import random
import logging
def enablePlayerMovement(self):
    self.accept('arrow_up', self.applyMovement, [Vec3(0, 100, 0)])
    self.accept('arrow_down', self.applyMovement, [Vec3(0, -100, 0)])
    self.accept('arrow_left', self.applyMovement, [Vec3(-100, 0, 0)])
    self.accept('arrow_right', self.applyMovement, [Vec3(100, 0, 0)])