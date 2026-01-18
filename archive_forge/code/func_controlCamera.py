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
def controlCamera(self, task):
    if self.mouseWatcherNode.hasMouse():
        mpos = self.mouseWatcherNode.getMouse()
        self.camera.setP(self.camera.getP() - mpos.getY() * 50)
        self.camera.setH(self.camera.getH() - mpos.getX() * 50)
    return task.cont